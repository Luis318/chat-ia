from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .serializers import ChatRequestSerializer
from .chat_service import get_answer

class ChatView(APIView):

    permission_classes = [IsAuthenticated]
    #permission_classes = [AllowAny] 

    def post(self, request):
        serializer = ChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        answer = get_answer(serializer.validated_data['question'])
        return Response({"answer": answer})
